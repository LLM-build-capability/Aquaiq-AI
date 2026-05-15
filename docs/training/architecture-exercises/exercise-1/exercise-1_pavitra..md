# Buying Coffee with a Contactless Card at a Café

## Introduction

This ArchiMate model represents a café contactless payment system used for buying coffee. The model is divided into three layers: Business Layer, Application Layer, and Technology Layer. These layers work together to show how customer activities, software services, and technical infrastructure interact during the coffee purchasing process.

## Business Layer

The Business Layer represents the activities and actors involved in the café process. The main business actors are the User and the Barista. The User performs the “Place and Order” business process, while the Barista performs the “Prepare Order” process. After the order is prepared, the payment process begins through the “Process Payment” business process. Once payment is completed, the system performs additional business activities such as “Issue Receipt,” “Earn Loyalty Points,” and “Update Inventory.”

Assignment relationships connect the User and Barista to their respective business processes. Flow relationships are used between “Place and Order,” “Process Payment,” “Issue Receipt,” and “Earn Loyalty Points” to represent the movement of information and business sequence. Triggering relationships are used where one process causes another process to happen, such as “Prepare Order” triggering “Update Inventory.”

## Application Layer

The Application Layer represents the software services and data management components that support the business activities. The “POS Interface Service” is responsible for customer interaction and accesses the “Menu Object” to display available coffee items. The “Order Management Service” handles customer orders and stores order information inside the “Order Database.” The “Payment Service” manages contactless card transactions and stores payment records inside the “Transaction Database.”

The “Member Service” handles loyalty point management and accesses the “Customer Database.” The “Inventory Service” updates stock availability using the “Inventory Database.” “Reporting and Analytics” collects operational data and triggers the “Notification Service.”

Access relationships are used between services and databases/data objects. Flow relationships are used between application services for data movement. Triggering relationships are used where one service starts another service. Serving relationships are used where application services support business processes.

## Technology Layer

The Technology Layer represents the physical devices, networks, and infrastructure that support the application services. The “Café POS Application Server” hosts important application services such as the POS Interface Service, Order Management Service, and Payment Service. The “POS Terminal” is used for placing orders, while the “Card Payment Terminal” handles contactless card transactions.

The “Barista Display” shows active orders to café staff, and the “Receipt Printer” prints receipts. The “WiFi/LAN Network” and “Local Network Switch” provide internal connectivity between devices and servers. The “Internet Gateway” enables external communication, and the “Cloud Backup Service” stores backup data.

Flow relationships are used between devices and network components. Serving relationships are used where the network infrastructure provides connectivity services. Realization relationships are used between the Café POS Application Server and the application services it hosts.

## Relationships Used

- Assignment relationship for connecting business actors and business processes.

- Flow relationship for process sequence and data movement.

- Access relationship between application services and databases/data objects.

- Triggering relationship where one process or service starts another process/service.

- Serving relationship where application services support business processes and infrastructure supports servers.

- Realization relationship between the Café POS Application Server and application services.

## Conclusion

This layered ArchiMate model demonstrates how business activities, application services, and technology infrastructure work together in a café contactless payment system. The model clearly represents customer interaction, payment handling, inventory management, loyalty management, reporting, and supporting technical infrastructure using proper ArchiMate elements and relationships.
 